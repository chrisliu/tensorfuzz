import numpy as np
def text_mutate(sentence, nofmutat, dictwords):
    #This function randomly select nofmutat locations from sentence. Then it randomly selects nofmutat dictwords and
    #mutates the original sentence using them
  # sentence : selected sentence to mutate
  # dictwords : 1D array ,word dictionary
  # nofmutat :  integer , number of words to be replaced
  wrdlen=len(sentence.split(" "))
  sentidx=np.random.randint(wrdlen, size=nofmutat)
  dictidx=np.random.randint(wrdlen, size=nofmutat)
  words=sentence.split(" ")
  tmp=""
  j=0
  for i in range(wrdlen):
    if i in sentidx:
      tmp=tmp + " " +lookp_dict[dictidx[j]]
      j+=1
    else:
      tmp=tmp + " " +words[i]
  return tmp