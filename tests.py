def test_data_split():
    """Tests train_test_split function"""
    from data import train_test_split

    data = list(range(100))
    train_data, test_data = train_test_split(data, 0.9)

    assert len(train_data) == 90
    assert len(test_data) == 10
    assert len(set(train_data).intersection(set(test_data))) == 0
