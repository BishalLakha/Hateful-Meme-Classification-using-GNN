import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from texthero import stopwords, preprocessing



def preprocessing_pipeline(text):

    #Call remove_stopwords and pass the custom_stopwords list
    default_stopwords = stopwords.DEFAULT
    #add a list of stopwords to the stopwords
    stop_w = ["I","It","The", "-","'", "."]
    custom_stopwords = default_stopwords.union(set(stop_w))

    # creating a custom pipeline to preprocess the raw text we have
    custom_pipeline = [preprocessing.lowercase
                    , preprocessing.fillna
                    , preprocessing.remove_diacritics
                    , preprocessing.remove_whitespace
                    , preprocessing.remove_angle_brackets
                    , preprocessing.remove_brackets
                    , preprocessing.remove_curly_brackets
                    , preprocessing.remove_html_tags
                    , preprocessing.remove_punctuation
                    ]
    
    # simply call clean() method to clean the raw text in ' comments' col and pass the custom_pipeline to pipeline argument
    text = hero.clean(text, pipeline = custom_pipeline)
    text = hero.remove_stopwords(text, custom_stopwords) 
    
    return comments


