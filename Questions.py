# -*- coding: utf-8 -*-

from pydantic import BaseModel

class Question(BaseModel):
    #sessionid: int
    questionid: int
    text: str
