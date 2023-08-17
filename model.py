import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Dense, LSTM, Multiply, Lambda, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras_tuner import RandomSearch
from sklearn.model_selection import StratifiedKFold
import keras.backend as K
import config

class ForexLSTMEncoder:
    def __init__(self):
        self.model = None
        self.context_vector_model = None

    def get_targets(self, data):
        """
        Generate binary targets based on average of bid and ask close prices.
        A positive price movement results in a target of 1 and negative or no movement results in 0.
        """
        avg_prices = (data[:, 3] + data[:, 7]) / 2 # Average of Ask and Bid Close price.
        price_diffs = np.diff(avg_prices)
        targets = np.where(price_diffs > 0, 1, 0)
        targets = np.append(targets, 0)
        return targets

    def hypermodel_build(self, hp, input_shape):
        inputs = Input(shape=input_shape)
        lstm_units = hp.Int('lstm_units', min_value=32, max_value=128, step=32)
        lstm_out, _, _ = LSTM(units=lstm_units, return_sequences=True, return_state=True, dropout=0.2)(inputs)

        attention_probs = Dense(config.Config.SEQ_LEN, activation='softmax', name='attention_probs')(lstm_out)
        attention_probs_expanded = Lambda(lambda x: K.expand_dims(x, -1))(attention_probs)
        attention_mul = Multiply(name='attention_multiply')([lstm_out, attention_probs_expanded])
        attention_mul_sum = Lambda(lambda x: K.sum(x, axis=1))(attention_mul)

        output = Dense(1, activation='sigmoid')(attention_mul_sum)

        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Print model summary
        model.summary()

        return model

    def create_dataset(self, X, time_steps=config.Config.SEQ_LEN):
        Xs = []
        for i in range(len(X) - time_steps):
            v = X[i:(i + time_steps)]
            Xs.append(v)
        return np.array(Xs)

    def train_model(self, X, y):
        input_shape = (config.Config.SEQ_LEN, config.Config.FEATURE_COUNT)

        tuner = RandomSearch(
            lambda hp: self.hypermodel_build(hp, input_shape),
            objective='val_accuracy',
            max_trials=25,  # Modify this based on your computational capability
            directory='forex_bot_hyperparameter_tuning',
            project_name='lstm_attention_model'
        )

        for trial in tuner.oracle.get_trials():
            if trial.status == "IDLE":
                tuner.oracle.update_trial(trial.trial_id, status="RUNNING")
                print("\n\nEvaluating set of hyperparameters:", trial.hyperparameters.values)

                # Build the model with the current set of hyperparameters
                model = tuner.hypermodel.build(trial.hyperparameters)

                # Use k-fold cross-validation to evaluate this set of hyperparameters
                avg_val_accuracy = self.kfold_evaluation(model, X, y)
                print(f"Average Validation Accuracy for current hyperparameters: {avg_val_accuracy:.4f}")

                # Inform the tuner of the results
                tuner.oracle.update_trial(trial.trial_id, metrics={'val_accuracy': avg_val_accuracy}, status="COMPLETED")

        # Get and print best hyperparameters
        best_hp = tuner.get_best_hyperparameters()[0]
        print("\n\nBest Hyperparameters Found:")
        for key, value in best_hp.values.items():
            print(f"{key}: {value}")

        # If you want to save and use the best model later
        self.model = tuner.get_best_models(num_models=1)[0]
        self.model.save(config.Config.MODEL_SAVE_PATH)


    def kfold_evaluation(self, model, data, targets, n_folds=5):
        """
        Evaluate a model's performance using k-fold cross-validation.
        """
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True)
        val_accuracies = []

        for fold_num, (train_idx, test_idx) in enumerate(kfold.split(data, targets), 1):
            print(f"Training on fold {fold_num}/{n_folds}...")

            X_train, X_test = data[train_idx], data[test_idx]
            y_train, y_test = targets[train_idx], targets[test_idx]

            history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=config.Config.EPOCHS, batch_size=config.Config.BATCH_SIZE)
            val_accuracies.append(history.history['val_accuracy'][-1])

            print(f"Fold {fold_num} Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")

        return np.mean(val_accuracies)


    def load_trained_model(self):
        """Load a previously trained model and extract its attention layer to be used for encoding."""
        try:
            self.model = load_model(config.Config.MODEL_SAVE_PATH)
            self.context_vector_model = Model(inputs=self.model.input, outputs=self.model.get_layer('attention_multiply').output)
        except Exception as e:
            print(f"Error loading trained model: {e}")

    def encode(self, data):
        """
        Encode data into a context vector using the attention mechanism.
        """
        if self.context_vector_model is None:
            raise Exception("Model not loaded. Load or train a model first.")

        try:
            return self.context_vector_model.predict(data)
        except Exception as e:
            print(f"Error encoding data: {e}")
            return None

    def prepare_and_train(self, data):
        """
        Prepare data by generating targets, handling lookahead bias, and splitting the dataset.
        Then proceed to train the model.
        """
        try:
            targets = self.get_targets(data)

            # Adjusting for lookahead bias
            data = np.roll(data, shift=1, axis=0)
            data = data[1:]
            targets = targets[:-1]

            # Split into train and validation sets
            split_index = int(0.8 * len(data))
            X_train, X_val = data[:split_index], data[split_index:]
            y_train, y_val = targets[:split_index], targets[split_index:]
            X_train = self.create_dataset(X_train)
            X_val = self.create_dataset(X_val)



            self.train_model(X_train, y_train, X_val, y_val)
        except Exception as e:
            print(f"Error preparing and training data: {e}")

