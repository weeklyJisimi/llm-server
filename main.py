import os

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
import openai
import json
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai.api_key = OPENAI_API_KEY

app = FastAPI()

class StyleAnalysisRequest(BaseModel):
    text1: str
    text2: str
    text3: str

class LetterGenerationRequest(BaseModel):
    style_characteristics: str
    purpose: str
    recipient: str
    episode: str

class LetterGenerationResponse(BaseModel):
    generated_letter: str = None
    additional_question: str = None

class AdditionalInfoRequest(BaseModel):
    question: str
    answer: str
    temperature: float = 0.7

@app.get("/test")
async def test():
    return {"response": "ok"}

async def call_gpt_api(prompt: str, temperature: float):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=temperature
        )
        return response.choices[0].message['content']
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_style")
async def analyze_style(request: StyleAnalysisRequest):
    prompt = f"""당신은 뛰어난 언어 분석가입니다. 아래 주어진 3가지 상황에 대한 텍스트를 분석하여 작성자의 문체적 특징을 추출해주세요. 다음 요소들에 주목해 주십시오:

1. 문장 길이와 구조
2. 어휘 선택 (격식체/비격식체, 현대어/고어 등)
3. 문법적 특징 (능동태/수동태 선호도, 접속사 사용 빈도 등)
4. 수사적 기법 (은유, 직유, 반복 등)
5. 전반적인 어조 (정중함, 친근함, 유머 등)
6. 상황에 따른 문체 변화

각 상황별 텍스트:

상황 1: {request.text1}
상황 2: {request.text2}
상황 3: {request.text3}

분석 결과를 바탕으로, 이 작성자의 전반적인 문체를 요약해서 요약문만 다음과 같은 JSON 형식으로 응답해주세요.

{{"style": "여기에 요약된 내용을 작성하세요"}}"""

    try:
        result = await call_gpt_api(prompt, temperature=0.7)
        return {"style_characteristics": result}
    except HTTPException as e:
        raise e

@app.post("/generate_letter", response_model=LetterGenerationResponse)
async def generate_letter(request: LetterGenerationRequest, response: Response):
    prompt = f"""당신은 사용자의 문체를 완벽히 모방할 수 있는 편지 작성 전문가입니다. 사용자의 문체를 정확히 반영한 편지를 작성하기 위해, 필요한 정보를 수집하고 편지를 생성할 것입니다.

1. 작성자의 문체 특징: {request.style_characteristics}
2. 편지 작성 목적: {request.purpose}
3. 편지 수신인과의 관계: {request.recipient}
4. 포함할 에피소드 또는 내용: {request.episode}

위 정보를 바탕으로, 편지 작성에 필요한 추가 정보가 있다면 사용자에게 한 가지 질문을 해주세요. 추가 정보가 필요 없다면 바로 편지를 작성해주세요.

만약 질문이 필요하다면, 다음과 같은 JSON 형식으로 응답해주세요:
{{"additional_question": "여기에 질문을 작성하세요"}}

질문이 필요 없다면, 다음과 같은 구조로 편지를 작성해 주세요:

1. 인사말
2. 도입부 (편지를 쓰게 된 계기나 근황)
3. 본문 (주요 내용과 에피소드 포함)
4. 마무리 (향후 계획이나 바람, 당부 등)
5. 맺음말

편지의 길이는 약 300-500자 정도로 작성해 주세요. 작성자의 문체를 정확히 반영하면서 자연스럽고 진정성 있는 편지를 작성해 주세요.

편지를 작성한 경우, 다음과 같은 JSON 형식으로 응답해주세요:
{{"generated_letter": "여기에 생성된 편지 내용을 작성하세요"}}"""

    try:
        result = await call_gpt_api(prompt, temperature=0.7)  # 추가: temperature 값 전달
        response_dict = json.loads(result)
        
        if "additional_question" in response_dict:
            response.status_code = 201
            return {"additional_question": response_dict["additional_question"]}
        elif "generated_letter" in response_dict:
            response.status_code = 200
            return {"generated_letter": response_dict["generated_letter"]}
        else:
            raise HTTPException(status_code=500, detail="Unexpected response format")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON response from API")
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/complete_letter", response_model=LetterGenerationResponse)
async def complete_letter(request: AdditionalInfoRequest, response: Response):
    question_answer_history = "\n".join([f"Q: {q}\nA: {a}" for q, a in zip(request.question_history, request.answer_history)])
    
    prompt = f"""이전에 다음과 같은 질문과 답변이 오갔습니다:

{question_answer_history}

가장 최근 질문에 대한 사용자의 새로운 답변: {request.new_answer}

이 모든 정보를 바탕으로, 편지를 작성해주세요. 다음과 같은 구조로 편지를 작성해 주세요:

1. 인사말
2. 도입부 (편지를 쓰게 된 계기나 근황)
3. 본문 (주요 내용과 에피소드 포함)
4. 마무리 (향후 계획이나 바람, 당부 등)
5. 맺음말

편지의 길이는 약 300-500자 정도로 작성해 주세요. 작성자의 문체를 정확히 반영하면서 자연스럽고 진정성 있는 편지를 작성해 주세요.

편지를 작성한 후, 다음과 같은 JSON 형식으로 응답해주세요:
{{"generated_letter": "여기에 생성된 편지 내용을 작성하세요"}}

만약 추가 정보가 여전히 부족하다면, 다음과 같은 JSON 형식으로 응답해주세요:
{{"additional_question": "여기에 추가 질문을 작성하세요", "answer_format": "여기에 예상되는 답변 형식을 작성하세요"}}"""

    try:
        result = await call_gpt_api(prompt, temperature=request.temperature)  # 추가: temperature 값 전달
        response_dict = json.loads(result)
        
        if "generated_letter" in response_dict:
            response.status_code = 200
            return {"generated_letter": response_dict["generated_letter"]}
        elif "additional_question" in response_dict:
            response.status_code = 201
            return {"additional_question": response_dict["additional_question"]}
        else:
            raise HTTPException(status_code=500, detail="Unexpected response format")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON response from API")
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
