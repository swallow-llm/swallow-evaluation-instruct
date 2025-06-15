# coding: utf-8
"""M-IFEval日本語指示判定ロジックのlighteval統合テスト用タスク"""

from lighteval.tasks.swallow.mifeval_ja.mifeval_ja_instructions_registry import JA_INSTRUCTION_DICT

def test_instruction_checker():
    # 例: 箇条書きCheckerを使ってみる
    checker_cls = JA_INSTRUCTION_DICT["detectable_format:number_bullet_lists"]
    checker = checker_cls("detectable_format:number_bullet_lists")
    desc = checker.build_description(num_bullets=2)
    print("指示文:", desc)
    # テスト用応答
    response = "・りんご\n・みかん"
    result = checker.check_following(response)
    print("判定結果:", result)

if __name__ == "__main__":
    test_instruction_checker()
