FIXED_DATE = "26 Jul 2024"


def patch_templates():
    from swift.llm.template.register import TEMPLATE_MAPPING
    m = TEMPLATE_MAPPING.get("llama3_2")
    if m is not None:
        head = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"Cutting Knowledge Date: December 2023\nToday Date: {FIXED_DATE}\n\n")
        m.prefix = [head + "<|eot_id|>"]
        m.system_prefix = [head + "{{SYSTEM}}<|eot_id|>"]


if __name__ == "__main__":
    from swift.cli.utils import try_use_single_device_mode
    try_use_single_device_mode()
    patch_templates()
    from swift.llm import sft_main
    sft_main()
