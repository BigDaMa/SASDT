__all__ = ["AbsPosPosition", "Position", "Propagate", "RegExPosPosition"]


def __getattr__(name):
    if name == "Propagate":
        from Propagate.propagate import Propagate

        return Propagate
    if name in {"AbsPosPosition", "Position", "RegExPosPosition"}:
        from Propagate.position import AbsPosPosition, Position, RegExPosPosition

        return {
            "AbsPosPosition": AbsPosPosition,
            "Position": Position,
            "RegExPosPosition": RegExPosPosition,
        }[name]
    raise AttributeError(name)
