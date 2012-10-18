import glob
import json


for a in glob.glob("params2-oneway-*.json"):
  data = json.load(open(a))
  if not data.get("correct"):
    continue
  f_old = a.replace("params2", "params")
  data_old = json.load(open(f_old))
  if not data.get("correct"):
     print "'correct' not found"  
  print "------",a
  print (data["correct"] - data_old["correct"])
