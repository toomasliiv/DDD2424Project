def train_student_model(student_model, teacher_model,X_train, y_train, X_val, y_val, hyperparams):
  tf.keras.backend.clear_session()
  file_location="/content/drive/My Drive/DD2424Files/Results/" + filename() + "/"
  metrics = ['accuracy']
  s_model_string = "student_model"
  t_model_string = "teacher_model"

  # Train teacher model
  BATCH_SIZE = hyperparams.get("batch_size")
  EPOCHS = hyperparams.get("epochs")
  

  save_callback = tf.keras.callbacks.ModelCheckpoint(
      file_location + "models_" + t_model_string, monitor='val_accuracy', verbose=0, save_best_only=True,
      save_weights_only=False, mode='max', save_freq='epoch')

  
  y_train = teacher_model.predict(X_train)
  print("shape of y_s_train: {} ".format(y_train.shape))
  student_model.compile(optimizer="adam", loss=tf.keras.losses.MSE, metrics= [tf.keras.metrics.CategoricalCrossentropy(from_logits=True)])
  history = student_model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val,y_val), verbose=1, callbacks = [save_callback])
  return student_model