import google.generativeai as genai
genai.configure(api_key="AIzaSyAKrO_ucEVGefsHBn7wcP_IMSBS3yqmA9E" )
for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(model.name)