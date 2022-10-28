from transformers import pipeline, set_seed

from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

# generator = pipeline("text-generation", model="EleutherAI/gpt-j-6B")
# # generator = pipeline("text-generation", model="EleutherAI/gpt-j-6B")
# # generator = pipeline("text-generation", model="EleutherAI/gpt-j-6B")
# set_seed(42)



mname = "facebook/blenderbot-400M-distill"
model = BlenderbotForConditionalGeneration.from_pretrained(mname)
tokenizer = BlenderbotTokenizer.from_pretrained(mname)

UTTERANCE = "My friends are cool but they eat too many carbs."
inputs = tokenizer([UTTERANCE], return_tensors="pt")
reply_ids = model.generate(**inputs)
print(tokenizer.batch_decode(reply_ids))

# all_chat = ""
# agent_temp = ""
# for i in range(3):
#     user_input = input("user: ")
#     all_chat += user_input + "\n"
#     while True:
#         agent_response = input("agent: ")
#         if "passs" in agent_response:
#             all_chat += "\n"
#             break
#         all_chat += " " + agent_response
#         suggestions = generator(
#             all_chat,
#             max_length=50,
#             num_return_sequences=3,
#             do_sample=True,
#             min_length=5
#             # top_k_top_p_filtering=0.9,
#             # TemperatureLogitsWarper=1,
#         )
#         l = len(all_chat)
#         for text in suggestions:
#             print("sugg : ", text["generated_text"][l:])
#         print("****************", all_chat)


