from keras.models import load_model


class Predictor(object):
    def __init__(self, model_path=None):
        print("Initializing neural network model from file: {}...".format(model_path))
        try:
            self.model = load_model(model_path)
        except OSError as e:
            print("[Failed]:", e)
        except TypeError:
            print("[Failed]: The model file has to be specified.")

    def predict(self):
        pass



if __name__ == '__main__':
    predictor = Predictor()