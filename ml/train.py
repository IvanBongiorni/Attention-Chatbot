
def train_model(model, params, X_train, Y_train, X_val, Y_val):
    '''
    Training Autograph function
    '''
    import numpy as np
    import tensorflow as tf
    
    @tf.function
    def train_on_batch():
        take = iteration * batch_size
        X_batch = X_train[ take:take+batch_size , : ]
        Y_batch = Y_train[ take:take+batch_size , : ]

        with tf.GrandientTape() as tape:
            current_loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(Y_batch, model(X_batch), 
                                                                from_logits = True))
        gradients = tape.gradient(current_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return current_loss
    
    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    
    loss_history = []
    
    for epoch in range(n_epochs):
        start = time.time()
        
        for iteration in range(X_train.shape[0] // batch_size):
            current_loss = train_on_batch()
            loss_history.append(current_loss)
    
        validation_loss = loss(Y_val, model(X_val))
        
        print('{}.   \tTraining Loss: {}   \tValidation Loss: {}   \tTime: {}ss'.format(
            epoch, 
            current_loss.numpy(),
            validation_loss.numpy(),
            round(time.time()-start, 2)
        ))
    print('Training complete.\n')
    model.save(params['save_path'] + )
    
    print('Model saved at:\n\t{}'.format(params['save_path'])
    return None

