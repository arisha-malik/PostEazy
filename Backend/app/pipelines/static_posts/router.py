import asyncio
import os
import uuid
import json
import urllib.parse
import httpx
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status, BackgroundTasks, Depends
from google import genai

from app.schemas import Platform, JobStatus, StaticPostScript, JobState
from app.services.ingest import extract_document_text
from app.jobs import job_store
from app.presets import get_preset
from app.core.pipeline_logging import log_pipeline_event

from dotenv import load_dotenv
load_dotenv()

router = APIRouter(prefix="/api/posts", tags=["Pipeline A: Static Posts"])

async def search_pixabay_image(query: str) -> str:
    """
    Searches Pixabay for a relevant stock photo using a short keyword query.
    Requires PIXABAY_API_KEY in .env.
    """
    api_key = os.getenv("PIXABAY_API_KEY", "").strip()
    if not api_key:
        return f"https://dummyimage.com/1080x1080/2a2a40/ffffff&text=Missing+Pixabay+Key"
        
    encoded_query = urllib.parse.quote(query)
    url = f"https://pixabay.com/api/?key={api_key}&q={encoded_query}&image_type=photo&orientation=horizontal&safesearch=true&per_page=3"
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            data = response.json()
            if data.get("hits") and len(data["hits"]) > 0:
                # Return the large image URL from the first result
                return data["hits"][0].get("largeImageURL", data["hits"][0].get("webformatURL"))
    except Exception as e:
        print(f"Pixabay search failed: {e}")
        
    return f"https://dummyimage.com/1080x1080/2a2a40/ffffff&text=No+Image+Found"

async def run_static_post_pipeline(job_id: str, text: str, platform: Platform):
    try:
        log_pipeline_event("static_posts", "pipeline_started", job_id=job_id, platform=platform.value, source_chars=len(text))
        
        job_store.update_job(job_id, status=JobState.RUNNING, stage="distilling_insights", progress=30)
        
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        preset = get_preset(platform)
        
        prompt = f"""You are an expert social media manager. Analyze this document.
        Create a highly engaging static carousel post script specifically for {platform}. 
        The tone should match: {preset.name}.
        
        You MUST return ONLY a raw JSON object exactly matching this structure. Do NOT wrap it in ```json blocks or backticks:
        {{
            "title": "A catchy title for the post",
            "slides": [
                {{
                    "layout_type": "hook",
                    "heading": "Strong opening hook",
                    "search_keywords": "Highly specific 3-5 word search query for a relevant stock photo (e.g., 'cascaded electronic amplifier circuit', not just 'electronics')"
                }},
                {{
                    "layout_type": "insight",
                    "heading": "A key takeaway from the document",
                    "search_keywords": "Highly specific 3-5 word search query representing this exact insight in a real-world scenario"
                }},
                {{
                    "layout_type": "cta",
                    "heading": "Call to action",
                    "search_keywords": "Highly specific 3-5 word search query for an image representing this exact action or growth"
                }}
            ],
            "caption": "The social media caption text",
            "hashtags": ["hashtag1", "hashtag2"]
        }}
        """
        
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
            contents=[text, prompt]
        )
        
        clean_json_str = response.text.replace('```json', '').replace('```', '').strip()
        script_data = json.loads(clean_json_str)

        # Stage 2: Slide Rendering using Pixabay
        job_store.update_job(job_id, stage="rendering_slides", progress=70, script=script_data)
        log_pipeline_event("static_posts", "script_ready", job_id=job_id, slides=len(script_data["slides"]), preset=preset.name)
        
        generated_slides = []
        for i, slide in enumerate(script_data["slides"]):
            query = slide.get("search_keywords", "abstract")
            image_url = await search_pixabay_image(query)
            slide["rendered_image"] = image_url
            generated_slides.append(slide)
            
        script_data["slides"] = generated_slides

        # Stage 3: Done
        output_urls = {
            "download_url": f"/api/posts/jobs/{job_id}/download"
        }
        job_store.update_job(
            job_id,
            status=JobState.DONE,
            stage="done",
            progress=100,
            output_urls=output_urls,
            script=script_data
        )
        log_pipeline_event("static_posts", "pipeline_completed", job_id=job_id, output_formats=["download"])
    except Exception as e:
        log_pipeline_event("static_posts", "pipeline_failed", job_id=job_id, level=40, error_summary=str(e)[:300])
        job_store.update_job(job_id, status=JobState.FAILED, stage="error", error=str(e))

@router.post("/scripts", response_model=StaticPostScript)
async def generate_post_script(
    file: UploadFile = File(...),
    platform: Platform = Form(Platform.LINKEDIN)
):
    raise HTTPException(status_code=501, detail="Use /jobs instead.")

@router.post("/jobs", response_model=JobStatus, status_code=status.HTTP_202_ACCEPTED)
async def create_static_post_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    platform: Platform = Form(Platform.LINKEDIN)
):
    doc_res = await extract_document_text(file)
    job = job_store.create_job(pipeline="static_posts", initial_stage="document_parsed")
    log_pipeline_event("static_posts", "source_ingested", job_id=job.job_id, filename=doc_res.filename, characters=doc_res.char_count, platform=platform.value)

    background_tasks.add_task(run_static_post_pipeline, job.job_id, doc_res.text, platform)

    return job

@router.get("/jobs/{job_id}", response_model=JobStatus)
async def get_static_post_job_status(job_id: str):
    job = job_store.get_job(job_id)
    if not job or job.pipeline != "static_posts":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Static post job '{job_id}' not found."
        )
    return job

@router.get("/jobs/{job_id}/download")
async def download_static_post_output(job_id: str):
    job = job_store.get_job(job_id)
    if not job or job.status != JobState.DONE:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Output not ready or job failed.")
    return {"message": f"Static post slides ready for job {job_id}"}
