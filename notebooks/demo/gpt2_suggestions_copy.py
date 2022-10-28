from transformers import pipeline, set_seed


from transformers import pipeline, set_seed

generator = pipeline("text-generation", model="gpt2")
set_seed(42)
all_chat = ""
agent_temp = ""
for i in range(3):
    user_input = input("user: ")
    all_chat += user_input + "\n"
    while True:
        agent_response = input("agent: ")
        if "passs" in agent_response:
            all_chat += "\n"
            break
        all_chat += " " + agent_response
        suggestions = generator(
            all_chat,
            max_length=10,
            num_return_sequences=3,
            # do_sample=True,
            # min_length=5
            # top_k_top_p_filtering=0.9,
            # TemperatureLogitsWarper=1,
        )
        l = len(all_chat)
        for text in suggestions:
            print("sugg : ", text["generated_text"][l:])
        print("****************", all_chat)
