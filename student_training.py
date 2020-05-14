# Train student model

def train_student_model(student_model, teacher_models, X_train, y_train, X_val, y_val, hyperparams, model_string):
  tf.keras.backend.clear_session()
  file_location="/content/drive/My Drive/DD2424Files/Results/" + filename() + "/"
  metrics = ['accuracy']

  # Train model
  BATCH_SIZE = hyperparams.get("batch_size")
  EPOCHS = hyperparams.get("epochs")
  

  save_callback = tf.keras.callbacks.ModelCheckpoint(
      file_location + "models_" + model_string, monitor='val_accuracy', verbose=0, save_best_only=True,
      save_weights_only=False, mode='max', save_freq='epoch')
  
  y_train = teacher_models[0].predict(X_train)

  for i in range(len(teacher_models)-1):
    y_train += teacher_models[i+1].predict(X_train)

  y_train /= len(teacher_models)

  print("shape of y_s_train: {} ".format(y_train.shape))
  student_model.compile(optimizer="adam", loss=tf.keras.losses.MSE, metrics= [tf.keras.metrics.CategoricalCrossentropy(from_logits=True), 'accuracy'])
  history = student_model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val,y_val), verbose=1, callbacks = [save_callback])
  
  # Save stuff
  plot_history(history, file_location + "acc_" + model_string + ".png", file_location + "loss_" + model_string + ".png")
  json.dump(history.history, open(file_location + "history_" + model_string, 'w'))
  json.dump(hyperparams, open(file_location + "hyperparams_" + model_string, 'w'))
  
  return student_model