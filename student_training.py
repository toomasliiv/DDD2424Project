def train_student_model(student_model, teacher_model, X_t_train, y_t_train, X_t_val, y_t_val, X_s_train, hyperparams):
  tf.keras.backend.clear_session()
  file_location="/content/drive/My Drive/DD2424Files/Results/" + filename() + "/"
  optimizer_teacher = tf.keras.optimizers.Adam(learning_rate=hyperparams.get("learning_rate"), epsilon=hyperparams.get("epsilon"))
  loss = tf.keras.losses.CategoricalCrossentropy()
  metrics = ['accuracy']
  s_model_string = "student_model"
  t_model_string = "teacher_model"
  teacher_model.add(tf.keras.layers.Activation("softmax"))
  teacher_model.compile(optimizer = optimizer_teacher, loss = loss, metrics = metrics)

  # Train teacher model
  BATCH_SIZE = hyperparams.get("batch_size")
  EPOCHS = hyperparams.get("epochs")
  

  save_callback = tf.keras.callbacks.ModelCheckpoint(
      file_location + "models_" + t_model_string, monitor='val_accuracy', verbose=0, save_best_only=True,
      save_weights_only=False, mode='max', save_freq='epoch')

  teacher_history = teacher_model.fit(X_t_train, y_t_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                          validation_data=(X_t_val, y_t_val), verbose=1, callbacks = [save_callback])
  
  y_s_train = teacher_model.predict(X_s_train)
  print("shape of y_s_train: {} shape of y_t_train: {}".format(y_s_train.shape, y_t_train.shape))
  student_model.compile(optimizer=optimizer_teacher, loss=tf.nn.softmax_cross_entropy_with_logits, metrics= [tf.keras.metrics.CategoricalCrossentropy()])
  student_model.fit(X_s_train, y_s_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_t_val,y_t_val), verbose=1, callbacks = [save_callback])
