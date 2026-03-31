"""Sample formatting utilities for encoder-mode inputs."""

from dataclasses import dataclass


PRESET_TEMPLATES = {
    "plain_sentence": "{sentence}",
    "word_sep_sentence": "{word} [SEP] {sentence}",
}


@dataclass
class SampleFormatter:
    """Formats each record into encoder input text via template placeholders."""

    template: str

    @classmethod
    def from_preset_or_template(cls, preset: str = "plain_sentence", template: str = None):
        if template:
            return cls(template=template)
        if preset not in PRESET_TEMPLATES:
            raise ValueError(f"Unknown preset: {preset}. Available: {sorted(PRESET_TEMPLATES.keys())}")
        return cls(template=PRESET_TEMPLATES[preset])

    def format(self, record: dict) -> str:
        values = {
            "id": record.get("id", ""),
            "word": record.get("lemma", ""),
            "lemma": record.get("lemma", ""),
            "sentence": record.get("sentence_text", ""),
            "phrase": record.get("sentence_text", ""),
            "pos": record.get("pos", ""),
        }
        try:
            return self.template.format(**values)
        except KeyError as exc:
            missing = str(exc)
            raise ValueError(f"Unknown template placeholder: {missing}")
