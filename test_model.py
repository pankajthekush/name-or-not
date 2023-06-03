from load_model import get_predicted_label

def test_valid_names():
    assert get_predicted_label('Jack Doroth') == 1
    assert get_predicted_label('Elon Musk') == 1

def test_valid_names_with_different_case():
    assert get_predicted_label('jack doroth') == 1
    assert get_predicted_label('eLoN mUsK') == 1

def test_invalid_names():
    assert get_predicted_label('John Doe') == 1
    assert get_predicted_label('Jane Smith') == 1

def test_website_headers():
    assert get_predicted_label('CEO of Google') == 0
    assert get_predicted_label('Founder of Facebook') == 0
    assert get_predicted_label('CTO at Microsoft') == 0

def test_website_designations():
    assert get_predicted_label('Software Engineer at Google') == 0
    assert get_predicted_label('Marketing Manager at Facebook') == 0
    assert get_predicted_label('Product Manager at Microsoft') == 0

def test_empty_string():
    assert get_predicted_label('') == 0

def test_special_characters_in_name():
    assert get_predicted_label('Elon Musk!') == 1

