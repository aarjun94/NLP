#!/usr/bin/env python
# coding: utf-8

# In[96]:


import re
import pandas as pd
from datetime import datetime, timedelta


# In[101]:


def find_dates(filename):
    month_names = {"Jan":"01", "Feb":"02", "Mar":"03", "Apr":"04", 
    "May":"05","Jun":"06", "Jul":"07", "Aug":"08", "Sep":"09", "Oct":"10", "Nov":"11", 
    "Dec":"12"}
    text1 = []
    with open(filename, 'r') as file:
        for line in file:
            
            match = re.search(r"(\d+)\t(.*)", line)
            id = match.group(1)
            text_org = match.group(2)
            
            match = re.search(r"\d+[/-]\d+(?:[/-]\d+)?|(?:\d+ )?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[.,]?(?:-\d+-\d+ | \d+(?:th|rd|st|nd)?,? \d+| \d+)|\d{4}", text_org)
            text = match.group()
            
            match = re.search(r"^((?:\d+ ))((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?))", text)
            if match:
                    x = match.group(1)
                    y = match.group(2)
                    match1 = re.search(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)", y)
                    text = re.sub(match.group(), f"{month_names[match1.group()]}/{x}/", text)
            
            match = re.search(r"^(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)", text)
            if match:
                    x = match.group()
                    match1 = re.search(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)", x)
                    text = re.sub(x, f"{month_names[match1.group()]}/", text)
            
            text = re.sub('\s','', text)
            
            match = re.search(r"^(\d+)[/-](\d+)$", text)
            if match:
                m = match.group(1)
                n = match.group(2)
                text = re.sub(match.group(), f"{m}/01/{n}", text)
                
            match = re.search(r"^(\d+)$", text)
            if match:
                y = match.group()
                text = re.sub(match.group(), f"01/01/{match.group()}", text)
                
            text = re.sub(r"[-]", "/", text)
            text = re.sub(r"[,]", "/", text)
            text = re.sub(r"[.]", "", text)
            
            match = re.search(r"^\d+[/-]\d+[/-](\d{2})$", text)
            if match:
                a = match.group(1)
                text = re.sub(match.group(1), f"19{match.group(1)}", text)
            
            match = re.search(r"^\d+[/-]\d+([/-]\d)$", text)
            if match:
                a = match.group(1)
                text = re.sub(match.group(1), "", text)
                
            match = re.search(r"^(\d+)[/][/](\d+)", text)
            if match:
                    x = match.group(1)
                    y = match.group(2)
                    text = re.sub(match.group(), f"{x}/01/{y}", text)
            
            match = re.search(r"^(\d+)[/](\d+)[/](\d+)", text)
            if match:
                    a = match.group(1)
                    b = match.group(2)
                    c = match.group(3)
                    text = re.sub(match.group(), f"{c}-{a}-{b}", text)
            
            match = re.search(r"^(\d{2})[/](\d+)", text)
            if match:
                    a = match.group(1)
                    b = match.group(2)
                    text = re.sub(match.group(), f"{b}-{a}-01", text)
                    
            
            match = re.search(r"^(\d{6})[-](\d+)[-]\d+", text)
            if match:
                    a = match.group(1)
                    b = match.group(2)
                    text = re.sub(match.group(), f"{a[2:6]}-{b}-{a[0:2]}", text)
                    
            match = re.search(r"(Marc), (\d{4})", line)
            if match:
                text = match.group()
                x = match.group(2)
                match1 = re.search(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)", text)
                text = re.sub(text, f"{x}-{month_names[match1.group()]}-01", text)
                
               # print(text)
            
            try:
                text = datetime.strptime(text,'%Y-%m-%d')
                new_date = text + timedelta(40)
            except:
                text = text
                new_date = text
            
            try: 
                text = text.strftime('%Y-%m-%d')
                new_date = new_date.strftime('%Y-%m-%d')
                
            except:
                text = text
            
            
            text1.append(id + '  ' + text +  " " + new_date)
            
            #print(id+ '\t' + text)
            
    with open('LHS712-Assg1-aarjun.txt', 'w') as outf:
        for x in text1:
                outf.write((x) + '\n')
            


# In[102]:


if __name__ == '__main__':
    x = find_dates('C:/Users/aarjun/Downloads/dates.txt')


# In[ ]:





# In[ ]:





# In[ ]:




