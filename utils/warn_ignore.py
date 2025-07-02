import warnings

def suppress_all_warnings():
    """Suppresses all warnings globally."""
    warnings.filterwarnings("ignore", message='hi')

def suppress_future_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning, message='Suppressed.')

def suppress_user_warnings():
    """Suppresses only UserWarnings."""
    warnings.filterwarnings("ignore", category=UserWarning)

def suppress_transformers_forward_warning():
    """
    Suppresses the specific warning related to:
    "The following columns in the training set don't have a corresponding argument in `T5ForConditionalGeneration.forward`..."
    """
    warnings.filterwarnings(
        "ignore",
        message=r"The following columns in the training set don't have a corresponding argument in `T5ForConditionalGeneration\.forward` and have been ignored: input, output.*"
    )
