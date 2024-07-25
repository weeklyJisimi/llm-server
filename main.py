import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
from dotenv import load_dotenv

load_dotenv()
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

app = FastAPI()

class StyleAnalysisRequest(BaseModel):
    text: str

class LetterGenerationRequest(BaseModel):
    style_characteristics: str
    purpose: str
    recipient: str
    episode: str

async def call_claude_api(prompt: str):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://api.anthropic.com/v1/completions",
                json={
                    "model": "claude-v1",
                    "prompt": prompt,
                    "max_tokens_to_sample": 1000
                },
                headers={
                    "Content-Type": "application/json",
                    "X-API-Key": CLAUDE_API_KEY
                }
            )
            response.raise_for_status()
            return response.json()["completion"]
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_style")
async def analyze_style(request: StyleAnalysisRequest):
    prompt = f"""당신은 문체 분석 전문가입니다. 주어진 텍스트를 분석하여 작성자의 고유한 문체 특성을 추출해주세요. 다음은 분석할 텍스트입니다:

{request.text}

위 텍스트에서 다음 요소들을 분석해주세요:
1. 문장 구조 (짧은 문장 선호 / 긴 문장 선호 / 복합문 사용 빈도 등)
2. 어휘 선택 (격식체 / 비격식체, 전문용어 사용 빈도, 관용구 사용 등)
3. 문체의 톤 (정중한 / 친근한 / 격식적인 등)
4. 특징적인 표현이나 문구
5. 기타 눈에 띄는 문체적 특징

분석 결과를 간결하게 요약해주세요."""

    try:
        result = await call_claude_api(prompt)
        return {"style_characteristics": result}
    except HTTPException as e:
        raise e

@app.post("/generate_letter")
async def generate_letter(request: LetterGenerationRequest):
    prompt = f"""당신은 개인화된 편지 작성 전문가입니다. 제공된 정보를 바탕으로 사용자의 문체와 의도에 맞는 편지를 작성해주세요.

문체 특성:
{request.style_characteristics}

편지 작성 요구사항:
- 목적: {request.purpose}
- 대상: {request.recipient}
- 포함할 에피소드: {request.episode}

위 정보를 바탕으로, 사용자의 문체 특성을 최대한 반영하여 자연스럽고 개인화된 편지를 작성해주세요. 편지의 길이는 약 300-500자로 해주세요."""

    try:
        result = await call_claude_api(prompt)
        return {"generated_letter": result}
    except HTTPException as e:
        raise e

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)