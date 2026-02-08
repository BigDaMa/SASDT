def is_valid_extraction(orig: str, ext: str, startidx: int, endidx: int) -> bool:
    if ext is None or ext == "":
        return False

    n = len(orig)

    if startidx is not None and startidx < 0:
        startidx += n
    if not endidx or endidx == -1:
        endidx = n
    elif endidx is not None and endidx < 0:
        endidx += n

    if startidx < 0 or endidx < 0 or startidx > n or endidx > n:
        return False
    if startidx >= endidx:
        return False

    if orig[startidx:endidx] != ext:
        return False

    if startidx == 0 and endidx == n:
        return True

    if startidx > 0 and orig[startidx - 1].isalnum() and orig[startidx].isalnum():
        return False

    if endidx < n and orig[endidx - 1].isalnum() and orig[endidx].isalnum():
        return False

    return True