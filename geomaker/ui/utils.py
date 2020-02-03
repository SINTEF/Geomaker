from string import ascii_lowercase

from PyQt5.QtCore import Qt


KEY_MAP = {
    Qt.Key_Space: 'SPC',
    Qt.Key_Escape: 'ESC',
    Qt.Key_Tab: 'TAB',
    Qt.Key_Return: 'RET',
    Qt.Key_Backspace: 'BSP',
    Qt.Key_Delete: 'DEL',
    Qt.Key_Up: 'UP',
    Qt.Key_Down: 'DOWN',
    Qt.Key_Left: 'LEFT',
    Qt.Key_Right: 'RIGHT',
    Qt.Key_Minus: '-',
    Qt.Key_Plus: '+',
    Qt.Key_Equal: '=',
}
KEY_MAP.update({
    getattr(Qt, 'Key_{}'.format(s.upper())): s
    for s in ascii_lowercase
})
KEY_MAP.update({
    getattr(Qt, 'Key_F{}'.format(i)): '<f{}>'.format(i)
    for i in range(1, 13)
})

def key_to_text(event):
    """Convert a Qt key event into a human-readable text format describing
    the key that was pressed. Standard Emacs format is used.
    """
    ctrl = event.modifiers() & Qt.ControlModifier
    shift = event.modifiers() & Qt.ShiftModifier

    try:
        text = KEY_MAP[event.key()]
    except KeyError:
        return

    if shift and text.isupper():
        text = 'S-{}'.format(text)
    elif shift:
        text = text.upper()
    if ctrl:
        text = 'C-{}'.format(text)

    return text


def angle_to_degrees(angle, directions):
    """Return a human-readable latitude or longitude representation of an
    angle in degrees, minutes and seconds of arc.
    """
    direction = directions[1 if angle > 0 else 0]
    angle = abs(angle)
    degrees, angle = divmod(angle, 1.0)
    minutes, angle = divmod(60 * angle, 1.0)
    seconds = 60 * angle
    return f"{degrees:.0f}° {minutes:.0f}' {seconds:.3f}'' {direction}"
