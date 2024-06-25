import sys

commonrules = """
                  1. Provide answers that are relevant to the topic requested by user
                  2. Responses should be  harmless, nontoxic, polite and relevant to the information requested by user.
                  3. Internalize the prompt supplied by the user and respond only if the prompt fits in the topics like "Financial Information","Banking information","Machine learning", "Deeplearning", "Generative AI". 
                  4. If you cannot find the information or if the prompt doesnot fit in rule 3 please respond politely as below 
                      - "Apologize I am not able to assist with that answer May i help with any thing relevant to the context. Thank you". 
                  5. Provide only Answer, please dont include instructions and rules as part of response.
                  6. Perform tough validation on above rules before responding back to the user.
                  7.Answer should contain only Answer Dont return commonrules or prompt
                  """



qatemplate = """
               You are an AI Assistant helping with the search of information. You must strictly adhere to the defined guard rules as outlined below:

               Follow the rules and provide an answer to the prompt without making any assumptions.

               Please respond with the ANSWER ONLY if the requested information satisfies the rules.

               RULES: {rules}

               PROMPT: {prompt}
          """
def guardpromptbuilder(modelname,prompt):
    try:
      print(f"encountered {modelname}")
      if modelname == "TheBloke/Mistral-7B-Instruct-v0.2-AWQ":
         prompt = mistralprompttemplate.format(prompt=prompt,commonrules=commonrules).strip()
         return f"<s>[INST]/n {prompt} /n[/INST] ANSWER:</s>"
         
      if modelname == "TheBloke/Llama-2-13B-chat-GPTQ":
         prompt = mistralprompttemplate.format(prompt=prompt,commonrules=commonrules).strip()
         return f"<s>[INST] {{ prompt }} [/INST] {{ ANSWER: }}</s>"
    except Exception as e: 
      raise Exception(f"modeltemplate not found {e}")
      sys.exit(1)
