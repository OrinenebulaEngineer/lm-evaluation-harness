from dataclasses import dataclass
from typing import Optional

@dataclass
class EvalRequest:
    """This class represents one evaluation request file."""

    model_arg: str
    json_filepath: str
    weight_type: str = "Original"
    model_type: Optional[str] = None  # pretrained, fine-tuned, etc. - define your own categories in
    precision: str = ""  # float16, bfloat16
    revision: str = "main"  # commit hash
    submitted_time: Optional[str] = (
        "2024"  # random date just so that we can still order requests by date
    )
    likes: Optional[int] = 0
    params: Optional[int] = None
    license: Optional[str] = ""
    base_model: Optional[str] = ""
    private: Optional[bool] = False
    status: str= None

    def get_model_args(self):
        """Edit this function if you want to manage more complex quantization issues. You'll need to map it to
        the evaluation suite you chose.
        """
        model_args = f"pretrained={self.model_arg},revision={self.revision}"

        if self.precision in ["float16", "bfloat16"]:
            model_args += f",dtype={self.precision}"

        # Quantized models need some added config, the install of bits and bytes, etc
        else:
            raise Exception(f"Unknown precision {self.precision}.")

        return model_args