import os

# JSON 형식의 데이터를 다루기 위해 사용됩니다. 사용자와 챗봇의 대화를 관리함으로써 대화 수 제한을 넘지 않도록 합니다.
import json

from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

# tiktoken 모듈은 OpenAI 모델이 텍스트를 토큰화하는 방식을 계산하기 위해 사용됩니다.
# API 요청 시 토큰 수를 예측하고 관리하는 데 필요합니다.
import tiktoken

load_dotenv(find_dotenv())

# os.environ["API_KEY"]를 통해 환경 변수에서 API_KEY라는 이름의 값을 가져옵니다.
# 이 API 키는 OpenAI API 서비스에 인증하기 위해 필요합니다.
API_KEY = os.environ["API_KEY"]

# 시스템 메시지는 챗봇의 역할, 성격, 응답 스타일 등을 정의하는 초기값입니다.
SYSTEM_MESSAGE = os.environ["SYSTEM_MESSAGE"]


BASE_URL = "https://api.together.xyz"
DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
FILENAME = "message_history.json"

# 최대 토큰 수를 제한합니다. 토큰 수 초과시, 오래된 대화가 우선적으로 삭제됩니다.
INPUT_TOKEN_LIMIT = 2048

# 챗봇 응답의 창의성을 결정합니다. 글쓰기 전문가이기 때문에 다소 높은 값으로 설정하였습니다.
DEFAULT_TEMPUERATURE = 0.7

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# OpenAI Chat API를 사용하여 채팅 응답을 생성하는 함수입니다.
def chat_completion(messages, model=DEFAULT_MODEL, temperature=DEFAULT_TEMPUERATURE, **kwargs):

    response = client.chat.completions.create(
        model=model,                 # 사용할 모델 지정
        messages=messages,           # 전체 대화 내역 전달
        temperature=temperature,     # 창의성 설정
        stream=False,                # 스트리밍 사용 안 함 (응답을 한 번에 받음)
        **kwargs,                    # 기타 추가 옵션 전달
    )

    # 메시지 내용을 추출하여 반환합니다.
    return response.choices[0].message.content

# 채팅 응답을 생성하고 출력하는 함수입니다.
def chat_completion_stream(messages, model=DEFAULT_MODEL, temperature=DEFAULT_TEMPUERATURE, **kwargs):

    # stream=True로 설정하여 스트리밍 응답을 요청합니다.
    response_stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=True, # 스트리밍 사용 (응답을 조각내어 순차적으로 받음)
        # 왜 스트리밍을 사용할까요? 사용자의 경험과 관련되어 있기 때문이죠. 응답을 조각내어 실시간으로 보여주면 사용자 입장에서는 "아! 응답하고 있구나" 라고 느낄 수 있어요.
        # 만약, 스트리밍을 사용하지 않을 경우, 응답이 전부 생성될 때까지 화면에 보여지지 않기 때문에 사용자 입장에서는 답답할 수 있겠지요.
        **kwargs,
    )

    response_content = "" # 전체 응답 내용에 대한 변수입니다.

    # 스트리밍 응답은 여러 개의 조각으로 나뉘어 전달됩니다.
    # 각 조각을 순회하면서 내용을 처리합니다.
    for chunk in response_stream:

        # 각 조각에서 텍스트 내용을 추출합니다.
        chunk_content = chunk.choices[0].delta.content
        if chunk_content is not None: # 내용이 있는 경우에만 처리합니다.
            print(chunk_content, end="") # 추출된 내용을 즉시 화면에 출력합니다.
            response_content += chunk_content 


    print() # 모든 조각 처리가 끝나면 줄바꿈을 해줍니다.
    return response_content # 완성된 전체 응답 내용을 반환합니다.

# 주어진 텍스트의 토큰 수를 계산하는 함수입니다.
def count_tokens(text, model):
    
    # "cl100k_base"는 GPT-3.5-turbo, GPT-4 등 최신 OpenAI 모델에서 사용하는 기본 인코딩 방식입니다.
    encoding = tiktoken.get_encoding("cl100k_base")

    # 텍스트를 인코딩하여 토큰 리스트로 변환합니다.
    tokens = encoding.encode(text)

    # 토큰 리스트의 길이를 반환하여 토큰 수를 얻습니다.
    return len(tokens)

# 메시지 목록 전체의 총 토큰 수를 계산하는 함수입니다.
def count_total_tokens(messages, model):
    
    total = 0 # 총 토큰 수를 저장할 변수를 초기화합니다.

    # 메시지 목록의 각 메시지에 대해 반복합니다.
    for message in messages:
        # 각 메시지의 'content' 키에 해당하는 값의 토큰 수를 계산하여 누적합니다.
        total += count_tokens(message.get("content", ""), model)
    return total # 누적된 총 토큰 수를 반환합니다.

# 메시지 목록의 총 토큰 수가 지정된 제한을 초과하지 않도록 오래된 메시지를 제거하는 함수입니다.
def enforce_token_limit(messages, token_limit, model=DEFAULT_MODEL):
    
    # 현재 총 토큰 수가 제한을 초과하는 동안 반복합니다.
    while count_total_tokens(messages, model) > token_limit:
        if len(messages) > 1: # 시스템 메시지 외에 다른 메시지가 남아있는 경우에만 제거를 시도합니다.
            messages.pop(1)
        else:
            # 시스템 메시지만 남았거나, 메시지가 없는 경우 더 이상 제거할 수 없으므로 반복을 중단합니다.
            break


# 파이썬 객체를 JSON 파일로 저장하는 함수입니다.
def save_to_json_file(obj, filename):
    try:
        # 인코딩 방식을 'utf-8'로 설정합니다.
        with open(filename, "w", encoding="utf-8") as file:

            # json.dump 함수를 사용하여 객체를 파일에 JSON 형식으로 씁니다.
            # indent=4는 JSON 파일을 사람이 읽기 쉽게 4칸 들여쓰기로 포맷팅합니다.
            json.dump(obj, file, indent=4, ensure_ascii=False)
    except IOError as e:
        print(f"'{filename}' 파일 저장 중 오류 발생: {e}")
    except Exception as e:
        print(f"알 수 없는 오류로 파일 저장 실패: {e}")

# JSON 파일에서 파이썬 객체를 불러오는 함수입니다.
def load_from_json_file(filename):

    # 파일 존재 여부를 먼저 확인합니다.
    if not os.path.exists(filename):
        # print(f"정보: '{filename}' 파일이 존재하지 않습니다. 새 대화 기록을 시작합니다.")
        return None # 파일이 없으면 None을 반환합니다.
    try:
        # 파일을 읽기 모드로 엽니다.
        with open(filename, "r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as e: 
        print(f"'{filename}' 파일 디코딩 중 오류 발생: {e}. 파일이 손상되었을 수 있습니다.")
        return None
    except IOError as e: 
        print(f"'{filename}' 파일 읽기 중 오류 발생: {e}")
        return None
    except Exception as e: 
        print(f"알 수 없는 오류로 파일 읽기 실패: {e}")
        return None

# 메인 챗봇을 실행하는 함수입니다.
def chatbot():
    # FILENAME에 지정된 파일에서 이전 대화 기록을 불러옵니다.
    messages = load_from_json_file(FILENAME)

    # 만약 불러온 대화 기록이 없거나(파일이 없거나, 읽기 실패) 비어있다면,
    # 시스템 메시지를 첫 번째 메시지로 하는 새로운 대화 목록을 생성합니다.
    if not messages:
        messages = [{"role": "system", "content": SYSTEM_MESSAGE}]

    # 채팅 시작 시 메시지를 출력합니다.
    print("Chatbot: 안녕하세요! 무엇을 도와드릴까요? (종료하려면 'quit' 또는 'exit'을 입력하세요.)")

    # 무한 루프를 통해 사용자와 계속 대화합니다.
    while True:
        try:
            # 사용자로부터 입력을 받습니다.
            user_input = input("You: ")
        except KeyboardInterrupt:
            print("\n사용자에 의해 중단되었습니다. 챗봇을 종료합니다.")
            break 

        # 사용자가 quit 또는 exit을 입력하면 대화를 종료합니다.
        if user_input.lower() in ['quit', 'exit']:
            print("챗봇을 종료합니다.")
            break 

        # 사용자의 입력을 메시지 목록에 추가합니다. 역할(role)은 user로 지정합니다.
        messages.append({"role": "user", "content": user_input})

        # 현재까지의 총 토큰 수를 계산하여 출력합니다.
        # 이 시점은 API 요청 전이므로, 토큰 제한 적용 전의 토큰 수입니다.
        # 메시지 목록이 토큰 제한을 초과하지 않도록 관리합니다.
        enforce_token_limit(messages, INPUT_TOKEN_LIMIT, DEFAULT_MODEL)

        # 토큰 제한 적용 후의 최종 토큰 수를 다시 계산하여 출력합니다.
        final_total_tokens = count_total_tokens(messages, DEFAULT_MODEL)
        print(f"[조정된 토큰 수: {final_total_tokens} / {INPUT_TOKEN_LIMIT}] (temperature: {DEFAULT_TEMPUERATURE})")

        # 토큰 제한을 왜 할까요?
        # 서비스를 구현할 때 토큰 제한을 하지 않으면 대답이 간결하지 않을 수 있습니다.
        # 간결하고 명확한 대답이 사용자 입장에서는 좋겠지요.
        # 또한, 토큰 제한을 하지 않으면 더 많은 컴퓨터 자원을 소모할 수 있습니다.
        # API 서비스는 토큰 수에 따라 요금을 부과하기 때문에 토큰을 제한하지 않으면 많은 비용이 발생할 수 있습니다.
        # 그래서 토큰 제한을 하는 것입니다.


        
        print("Chatbot: ", end="")

        # 챗봇의 응답을 스트리밍 형태로 받기 위해 chat_completion_stream 함수를 호출합니다. 
        response = chat_completion_stream(messages, temperature=DEFAULT_TEMPUERATURE)

        # API로부터 받은 챗봇의 응답이 비어있지 않다면, 메시지 목록에 추가합니다.
        if response: # 응답이 성공적으로 생성된 경우
            messages.append({"role": "assistant", "content": response})
            # 업데이트된 전체 대화 기록을 JSON 파일에 저장합니다.
            save_to_json_file(messages, FILENAME)
        else: # 응답 생성에 실패한 경우
            # 오류가 발생했을 때, 마지막 사용자 메시지를 제거하여 동일한 오류가 반복되는 것을 방지할 수 있습니다. (선택적 처리)
            if messages and messages[-1]["role"] == "user":
                messages.pop()
            print("Chatbot: 죄송합니다. 응답을 생성하는 데 실패했습니다. 다시 시도해 주세요.")


if __name__ == "__main__":
    # API 키가 환경 변수에 설정되어 있는지 확인합니다.
    if not API_KEY:
        print("치명적 오류: .env 파일에서 API_KEY를 찾을 수 없습니다. API_KEY 환경 변수를 설정해주세요. 프로그램을 종료합니다.")
        exit() 

    chatbot()
