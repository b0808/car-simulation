from utiles import *
from sklearn.model_selection import train_test_split


path = "data-master"
data = importdata(path)

data  = balancedata(data,display=False)

imagepath , stearing = loaddata(path,data)
print(imagepath[0],stearing[0])
xtrain ,xval,ytrain,yval = train_test_split(imagepath,stearing,test_size=0.2,random_state=42)
print(len(xtrain),len(xval))



model=Model()

model.summary()
history= model.fit(batchgen(xtrain,ytrain,100,1),steps_per_epoch=300,epochs=10,
          validation_data = batchgen(xval,yval,100,0),validation_steps=200)


model.save('model2.h5')
print('model Saved')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['traning','validation'])

plt.title('Loss')
plt.xlabel('Epoch')
plt.show()