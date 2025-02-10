"""Exceptions for inspectors"""


class MissingNetParam(Exception):
    pass


class MissingMultipleDataParams(Exception):
    pass


class MissingDataParam(Exception):
    pass


class MissingTrainerSpec(Exception):
    pass


class MissingTesterSpec(Exception):
    pass


class UnequalNetParamWarning(Warning):
    pass


class InvalidReturnType(Exception):
    pass
