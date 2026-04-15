from models import BATM, ETM, GMNTM, GSM, WTM


MODEL_REGISTRY = {
    "gsm": GSM,
    "wtm": WTM,
    "etm": ETM,
    "gmntm": GMNTM,
    "batm": BATM,
}


def get_model_class(model_name):
    return MODEL_REGISTRY[model_name.lower()]
