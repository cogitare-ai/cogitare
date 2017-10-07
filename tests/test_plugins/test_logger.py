from cogitare.plugins import Logger
from tqdm import tqdm
import mock


def test_logger1(capsys):
    tqdm._instances = set()
    l = Logger()
    l(loss_mean=1.0)
    out, err = capsys.readouterr()
    assert '[Logger] Loss: 1.000000 ' in err


def test_logger2(capsys):
    l = Logger(title='[[Test]]', msg='Msg: {loss}', show_time=False)
    l(loss=1.0)
    out, err = capsys.readouterr()
    assert '[[Test]] Msg: 1.0' in err


def test_logger_file(capsys):
    f = mock.Mock()
    f.write = mock.MagicMock(return_value=None)

    l = Logger(output_file=f)
    l(loss_mean=1.0)
    out, err = capsys.readouterr()
    assert '[Logger] Loss: 1.000000 ' in err
    assert f.write.called


def test_logger_with_tqdm(capsys):
    bar = tqdm(total=10)
    bar.update(3)

    l = Logger()
    l(loss_mean=1.0)
    out, err = capsys.readouterr()
    assert '[Logger] Loss: 1.000000 ' in err
