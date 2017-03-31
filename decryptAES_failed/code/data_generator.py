__author__ = 'milad'
from cryptography.hazmat.primitives.ciphers import Cipher,algorithms,modes
from cryptography.hazmat.backends import default_backend
import struct
import os
import cPickle as pickle
backend=default_backend()
key=struct.pack('@16s','milad')

cipher= Cipher(algorithms.AES(key),modes.ECB(),backend=backend)
encryptor=cipher.encryptor()

with open('output','w') as f:
    mydata={}
    for i in xrange(10000000):
        plain=os.urandom(16)
        cip=encryptor.update(plain)
        mydata[plain]=cip
    pickle.dump(mydata,f)




