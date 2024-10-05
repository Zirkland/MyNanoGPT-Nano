import torch
from transformers import GPT2Tokenizer
from MyTransformer import MyModel

# 加载GPT-2分词器
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# 初始化自定义模型并加载训练好的权重
model = MyModel()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 加载训练好的模型权重
model.load_state_dict(torch.load('trained_model.pth', map_location=device))


def generate_text(prompt, max_length=100, num_return_sequences=1):
    # 将模型设置为评估模式
    model.eval()

    # 对输入的提示词进行编码
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    generated_texts = []
    for _ in range(num_return_sequences):
        output = input_ids
        for _ in range(max_length):
            with torch.no_grad():
                logits = model(output)
                next_token_logits = logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                output = torch.cat((output, next_token_id), dim=1)
                if next_token_id.item() == tokenizer.eos_token_id:
                    break
        generated_texts.append(tokenizer.decode(output[0], skip_special_tokens=True))

    return generated_texts


# 示例用法
prompt = "https"
generated_texts = generate_text(prompt, max_length=50, num_return_sequences=3)

for i, text in enumerate(generated_texts):
    print(f"Generated Text {i + 1}: {text}")
