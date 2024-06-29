import torch
from transformers import T5Tokenizer, GPT2LMHeadModel

# モデルとトークナイザーのロード
model_path =   
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# モデルを評価モードに設定
model.eval()

def generate_answer(question, max_length=100, temperature=0.7, top_k=50, top_p=0.9):
    # 質問をフォーマットしてトークン化
    input_text = f"<Human>: {question}\n<AI>: "
    inputs = tokenizer.encode(input_text, return_tensors='pt')
    
    # モデルに入力してテキストを生成
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            do_sample=True,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 生成されたテキストをデコード
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 元の入力部分を取り除いて回答部分のみを取得
    answer = answer[len(input_text):].strip()
    return answer

# ユーザーから質問を入力して回答を生成するループ
while True:
    question = input("質問を入力してください (終了する場合は 'exit' を入力): ")
    if question.lower() == 'exit':
        print("プログラムを終了します。")
        break
    
    # 初回の回答生成
    answer = generate_answer(question)
    
    while True:
        print("回答:", answer)
        feedback = input("この回答で問題がありますか？ (y/n): ").strip().lower()
        
        if feedback == 'y':
            print("再生成します...")
            answer = generate_answer(question)
        elif feedback == 'n':
            break
        else:
            print("正しい入力をしてください (y/n)")
    
    print()  # 改行して次の質問を促します