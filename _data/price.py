def item_price(shop_file, item):
    price_dict = shopping(shop_file)
    if item in price_dict:
        return price_dict[item]
    else:
        return f"{item}은(는) 해당 상점에서 판매하지 않습니다."


def shopping(filename):
    # 상품명과 가격을 저장할 딕셔너리 초기화
    price_dict = {}

    # 파일 읽어서 딕셔너리에 추가
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            items = line.strip().split()
            # 상품명과 가격 추출
            if len(items) == 2:  # 공백 문자를 포함하는 경우에만 추출
                item, price_str = items
                if '원' in price_str:  # 가격 문자열에 '원'이 있는 경우에만 추출
                    price = int(price_str[:-1].replace(',', ''))  # 가격 문자열에서 '원'과 ','를 제외하고 정수형으로 변환
                    # 딕셔너리에 추가
                    price_dict[item] = price

    return price_dict
