

'''
model.fit(
    #[trainAttrX, trainImagesX], trainY,
    X_train, y_train,
    #validation_data=([testAttrX, testImagesX], testY),
    validation_data=(X_test, y_test),
    epochs=200, batch_size=8)

# make predictions on the testing data
print("[INFO] predicting masks...")
preds = model.predict(X_test)

'''


model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(np.array(X_train), np.array(y_train), epochs=20, batch_size=1)#
score = model.evaluate(X_test, y_test)

print(score)



# create the MLP and CNN models

cnn = models.create_cnn(512, 512, 3, regress=False)

# our final FC layer head will have two dense layers, the final one
# being our regression head

x = Dense(4, activation="relu")(cnn.output)
x = Dense(1, activation="linear")(x)

# our final model will accept categorical/numerical data on the MLP
# input and images on the CNN input, outputting a single value (the
# predicted price of the house)
model = Model(inputs=[cnn.input], outputs=x)


# compile the model using mean absolute percentage error as our loss,
# implying that we seek to minimize the absolute percentage difference
# between our price *predictions* and the *actual prices*
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)
 
# train the model
print("[INFO] training model...")

print("printing items")
for index, item in enumerate(X_train):
    print("item was", index)
    print(item)

print(model.summary())
