from typing import List, Optional, Callable, Literal, Dict, Any
import re

def remove_instruction_decorator(func):
    """
    関数の返り値（Docオブジェクト）の instruction 属性を None にするデコレータ
    """
    def wrapper(*args, **kwargs):
        doc = func(*args, **kwargs)
        doc.instruction = None
        return doc
    return wrapper

def _regex_extractor(obj_regex, text: str, 
                     match_group_name: str,
                     extraction_mode: Literal["first_match", "last_match", "any_match"]) -> List[str]:
    """
    正規表現とテキストを渡すと extraction mode に従って抽出結果を返す関数．
    - extraction_mode="first_match": 最初のマッチのみを返す
    - extraction_mode="last_match": 最後のマッチのみを返す
    - extraction_mode="any_match": すべてのマッチを返す    
    """
    
    lst_matches_with_positions = [(match.group(match_group_name), match.start(), match.end()) for match in obj_regex.finditer(text)]
    # 出現箇所の昇順にソート．つまり最後に出現したものが配列の最後に入っている
    lst_matches_with_positions = sorted(lst_matches_with_positions, key=lambda x: (x[2], -x[1]), reverse=False)
    
    if len(lst_matches_with_positions) == 0:
        return []
    else:
        if extraction_mode == "first_match":
            return [lst_matches_with_positions[0][0]]
        elif extraction_mode == "last_match":
            return [lst_matches_with_positions[-1][0]]
        elif extraction_mode == "any_match":
            return [match[0] for match in lst_matches_with_positions]
