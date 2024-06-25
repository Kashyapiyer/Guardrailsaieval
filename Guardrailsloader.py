from typing import Optional
from huggingface_hub.inference._generated.types import object_detection
from transformers import AutoModelForCausalLM, AutoTokenizer
from helperutil import read_yaml_conf,intializelogger
from guardrails.hub import (
    ToxicLanguage,
    RegexMatch,
    GibberishText,
    DetectPII,
    NSFWText,
    FinancialTone,
    ProfanityFree, 
    SensitiveTopic, 
    RestrictToTopic
)
from guardrails import Guard
#from prompttemplate import guardpromptbuilder
import torch
import os,json,sys
import logging 

class Guardloader:


  def __init__(self,logger, configfile, contextinp:str, validation_args="full") -> None:
      self.guardconf = read_yaml_conf(configfile)
      self.contextinp = contextinp
      self.logger = logger
      self.validation_args = validation_args

  def finalresonse(self,result):
      if result.contains("Validation failed for field with errors:"):
          validationresult = False
      else:
          validationresult = True 
      return result
      
  def evalrunner(self):
      self.allowedvalidations = self.guardconf["allowedvalidations"]

      if self.validation_args not in self.allowedvalidations:
            raise ValueError(f'selected {self.validation_args} not allowed. please select from {self.allowedvalidations}')
      
      try:
           response={}
           if self.validation_args == "deterministic":
              result = self.deterministiceval(self.contextinp)
              response['deterministic'] = result
              return response 

           elif self.validation_args == "probablistic":
              result = self.probablisticeval(self.contextinp)
              response['probablistic']= result 
              return response 
           else:    
              response['deterministic'] = self.deterministiceval(self.contextinp)
              self.logger.info("Proceeding with probablisticeval")
              response['probablistic']= self.probablisticeval(self.contextinp)         
              return response 
      except Exception as errmsg:
             self.logger.error(f'encountered issue while evaluating {self.validation_args}')
             print(f'encountered issue while evaluating {self.validation_args}')
             return errmsg
          

  def deterministiceval(self, contextinp):
          """
          input: Accepts sentence input and performs relevant deterministic evaluvation checks 
          return json repsonse

          """
          try:
              self.pii_entities = self.guardconf['pii_entities']
              guard = Guard().use_many(
                  ToxicLanguage(
                      validation_method='sentence',
                      threshold=0.5
                  ),
                  RegexMatch(
                  regex="^[A-Z].*",
                  on_fail="exception"
                  ),
                  GibberishText(
                      threshold=0.5,
                      validation_method="sentence",
                      on_fail="exception"
                  ),
                  DetectPII(
                      pii_entities=self.pii_entities,
                      on_fail="exception"
                    ),
                  NSFWText(
                      threshold=0.8, validation_method="sentence", on_fail="exception"
                  ),
                  ProfanityFree(on_fail="exception")

              )
              response = guard.validate(contextinp)
              response = json.loads(response.json())
              return response
          except Exception as errmsg:
              self.logger.error("at error block")
              response = errmsg.args[0]
              return response


  def probablisticeval(self, contextinp):
          """
          input: Accepts context input and performs relevant probalistic evaluvation checks 
          return json repsonse

          """
          try:
              # read the list of restricted from conf 
              self.restrictedtopics = self.guardconf['restrictedtopics']
              self.senstivetopics = self.guardconf['senstivetopics']

              guard = Guard().use_many(

                        

                        RestrictToTopic(
                                        valid_topics=self.restrictedtopics,
                                        disable_classifier=True,
                                        disable_llm=True,
                                        on_fail="exception"
                                      )
              )
              response = guard.validate(contextinp)
              response = json.loads(response.json())
              return response
          except Exception as errmsg:
              self.logger.error("at error block")
              response = errmsg.args[0]
              return response
    