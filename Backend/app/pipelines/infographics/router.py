import asyncio
import os
import uuid
import json
import io
import base64
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, status
from google import genai
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from app.schemas import JobStatus, JobState
from app.jobs import job_store
from app.config import settings

router = APIRouter(prefix="/api/infographics", tags=["Pipeline D: Infographics"])

def render_chart_to_base64(chart_data: dict) -> str:
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    c_type = chart_data.get("type", "bar")
    labels = chart_data.get("labels", [])
    values = chart_data.get("values", [])
    title = chart_data.get("title", "Chart")
    
    if c_type == "bar":
        sns.barplot(x=labels, y=values, palette="viridis")
        plt.xlabel(chart_data.get("x_label", ""))
        plt.ylabel(chart_data.get("y_label", ""))
    elif c_type == "line":
        sns.lineplot(x=labels, y=values, marker="o", color="b")
        plt.xlabel(chart_data.get("x_label", ""))
        plt.ylabel(chart_data.get("y_label", ""))
    elif c_type == "pie":
        plt.pie(values, labels=labels, autopct='%1.1f%%', colors=sns.color_palette("pastel"))
    
    plt.title(title)
    plt.tight_layout()
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    b64_str = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()
    
    return f"data:image/png;base64,{b64_str}"

async def run_infographic_pipeline(job_id: str, file_path: str):
    try:
        job_store.update_job(job_id, status=JobState.RUNNING, stage="extracting_data", progress=30)
        
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        gemini_file = await asyncio.to_thread(client.files.upload, file=file_path)
        
        prompt = """You are an expert data analyst and visualization designer. Analyze this document (it could be a PDF, image, or text).
        Extract any numerical data, trends, comparisons, or structured information that can be visualized as an infographic.
        Identify up to 3 distinct charts that can be generated.

        You MUST return ONLY a raw JSON object exactly matching this structure. Do NOT wrap it in ```json blocks:
        {
            "title": "Title of the Infographic Report",
            "summary": "A brief 2-sentence summary of the extracted data insights.",
            "charts": [
                {
                    "type": "bar",
                    "title": "Title of the chart",
                    "x_label": "X-axis label (e.g., Year, Category)",
                    "y_label": "Y-axis label (e.g., Revenue, Percentage)",
                    "labels": ["Label 1", "Label 2", "Label 3"],
                    "values": [10, 20, 30]
                }
            ]
        }
        """
        
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
            contents=[gemini_file, prompt]
        )
        
        clean_json_str = response.text.replace('```json', '').replace('```', '').strip()
        script_data = json.loads(clean_json_str)
        
        job_store.update_job(job_id, stage="rendering_charts", progress=70, script=script_data)
        
        generated_charts = []
        for chart in script_data.get("charts", []):
            try:
                chart_img = await asyncio.to_thread(render_chart_to_base64, chart)
                chart["rendered_image"] = chart_img
            except Exception as e:
                print(f"Failed to render chart: {e}")
                chart["rendered_image"] = None
            generated_charts.append(chart)
            
        script_data["charts"] = generated_charts
        
        job_store.update_job(
            job_id,
            status=JobState.DONE,
            stage="done",
            progress=100,
            script=script_data
        )
    except Exception as e:
        job_store.update_job(job_id, status=JobState.FAILED, stage="error", error=str(e))

@router.post("/jobs", response_model=JobStatus, status_code=status.HTTP_202_ACCEPTED)
async def create_infographic_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    job = job_store.create_job(pipeline="infographic", initial_stage="extracting_data")
    ext = os.path.splitext(file.filename)[1]
    temp_path = os.path.join(settings.WORK_DIR, f"{job.job_id}{ext}")
    
    with open(temp_path, "wb") as f:
        f.write(await file.read())
        
    background_tasks.add_task(run_infographic_pipeline, job.job_id, temp_path)
    
    return job

from fastapi import HTTPException
@router.get("/jobs/{job_id}", response_model=JobStatus)
async def get_infographic_job_status(job_id: str):
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Infographic job '{job_id}' not found."
        )
    return job
