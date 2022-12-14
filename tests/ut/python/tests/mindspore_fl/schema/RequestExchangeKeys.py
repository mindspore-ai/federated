# automatically generated by the FlatBuffers compiler, do not modify

# namespace: schema

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class RequestExchangeKeys(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = RequestExchangeKeys()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsRequestExchangeKeys(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    # RequestExchangeKeys
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # RequestExchangeKeys
    def FlId(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # RequestExchangeKeys
    def CPk(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 1))
        return 0

    # RequestExchangeKeys
    def CPkAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint8Flags, o)
        return 0

    # RequestExchangeKeys
    def CPkLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # RequestExchangeKeys
    def CPkIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        return o == 0

    # RequestExchangeKeys
    def SPk(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 1))
        return 0

    # RequestExchangeKeys
    def SPkAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint8Flags, o)
        return 0

    # RequestExchangeKeys
    def SPkLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # RequestExchangeKeys
    def SPkIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        return o == 0

    # RequestExchangeKeys
    def Iteration(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # RequestExchangeKeys
    def Timestamp(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # RequestExchangeKeys
    def IndIv(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 1))
        return 0

    # RequestExchangeKeys
    def IndIvAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint8Flags, o)
        return 0

    # RequestExchangeKeys
    def IndIvLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # RequestExchangeKeys
    def IndIvIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        return o == 0

    # RequestExchangeKeys
    def PwIv(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 1))
        return 0

    # RequestExchangeKeys
    def PwIvAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint8Flags, o)
        return 0

    # RequestExchangeKeys
    def PwIvLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # RequestExchangeKeys
    def PwIvIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        return o == 0

    # RequestExchangeKeys
    def PwSalt(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 1))
        return 0

    # RequestExchangeKeys
    def PwSaltAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint8Flags, o)
        return 0

    # RequestExchangeKeys
    def PwSaltLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # RequestExchangeKeys
    def PwSaltIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        return o == 0

    # RequestExchangeKeys
    def Signature(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 1))
        return 0

    # RequestExchangeKeys
    def SignatureAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint8Flags, o)
        return 0

    # RequestExchangeKeys
    def SignatureLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # RequestExchangeKeys
    def SignatureIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        return o == 0

    # RequestExchangeKeys
    def CertificateChain(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(22))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.String(a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return ""

    # RequestExchangeKeys
    def CertificateChainLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(22))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # RequestExchangeKeys
    def CertificateChainIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(22))
        return o == 0

def Start(builder): builder.StartObject(10)
def RequestExchangeKeysStart(builder):
    """This method is deprecated. Please switch to Start."""
    return Start(builder)
def AddFlId(builder, flId): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(flId), 0)
def RequestExchangeKeysAddFlId(builder, flId):
    """This method is deprecated. Please switch to AddFlId."""
    return AddFlId(builder, flId)
def AddCPk(builder, cPk): builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(cPk), 0)
def RequestExchangeKeysAddCPk(builder, cPk):
    """This method is deprecated. Please switch to AddCPk."""
    return AddCPk(builder, cPk)
def StartCPkVector(builder, numElems): return builder.StartVector(1, numElems, 1)
def RequestExchangeKeysStartCPkVector(builder, numElems):
    """This method is deprecated. Please switch to Start."""
    return StartCPkVector(builder, numElems)
def AddSPk(builder, sPk): builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(sPk), 0)
def RequestExchangeKeysAddSPk(builder, sPk):
    """This method is deprecated. Please switch to AddSPk."""
    return AddSPk(builder, sPk)
def StartSPkVector(builder, numElems): return builder.StartVector(1, numElems, 1)
def RequestExchangeKeysStartSPkVector(builder, numElems):
    """This method is deprecated. Please switch to Start."""
    return StartSPkVector(builder, numElems)
def AddIteration(builder, iteration): builder.PrependInt32Slot(3, iteration, 0)
def RequestExchangeKeysAddIteration(builder, iteration):
    """This method is deprecated. Please switch to AddIteration."""
    return AddIteration(builder, iteration)
def AddTimestamp(builder, timestamp): builder.PrependUOffsetTRelativeSlot(4, flatbuffers.number_types.UOffsetTFlags.py_type(timestamp), 0)
def RequestExchangeKeysAddTimestamp(builder, timestamp):
    """This method is deprecated. Please switch to AddTimestamp."""
    return AddTimestamp(builder, timestamp)
def AddIndIv(builder, indIv): builder.PrependUOffsetTRelativeSlot(5, flatbuffers.number_types.UOffsetTFlags.py_type(indIv), 0)
def RequestExchangeKeysAddIndIv(builder, indIv):
    """This method is deprecated. Please switch to AddIndIv."""
    return AddIndIv(builder, indIv)
def StartIndIvVector(builder, numElems): return builder.StartVector(1, numElems, 1)
def RequestExchangeKeysStartIndIvVector(builder, numElems):
    """This method is deprecated. Please switch to Start."""
    return StartIndIvVector(builder, numElems)
def AddPwIv(builder, pwIv): builder.PrependUOffsetTRelativeSlot(6, flatbuffers.number_types.UOffsetTFlags.py_type(pwIv), 0)
def RequestExchangeKeysAddPwIv(builder, pwIv):
    """This method is deprecated. Please switch to AddPwIv."""
    return AddPwIv(builder, pwIv)
def StartPwIvVector(builder, numElems): return builder.StartVector(1, numElems, 1)
def RequestExchangeKeysStartPwIvVector(builder, numElems):
    """This method is deprecated. Please switch to Start."""
    return StartPwIvVector(builder, numElems)
def AddPwSalt(builder, pwSalt): builder.PrependUOffsetTRelativeSlot(7, flatbuffers.number_types.UOffsetTFlags.py_type(pwSalt), 0)
def RequestExchangeKeysAddPwSalt(builder, pwSalt):
    """This method is deprecated. Please switch to AddPwSalt."""
    return AddPwSalt(builder, pwSalt)
def StartPwSaltVector(builder, numElems): return builder.StartVector(1, numElems, 1)
def RequestExchangeKeysStartPwSaltVector(builder, numElems):
    """This method is deprecated. Please switch to Start."""
    return StartPwSaltVector(builder, numElems)
def AddSignature(builder, signature): builder.PrependUOffsetTRelativeSlot(8, flatbuffers.number_types.UOffsetTFlags.py_type(signature), 0)
def RequestExchangeKeysAddSignature(builder, signature):
    """This method is deprecated. Please switch to AddSignature."""
    return AddSignature(builder, signature)
def StartSignatureVector(builder, numElems): return builder.StartVector(1, numElems, 1)
def RequestExchangeKeysStartSignatureVector(builder, numElems):
    """This method is deprecated. Please switch to Start."""
    return StartSignatureVector(builder, numElems)
def AddCertificateChain(builder, certificateChain): builder.PrependUOffsetTRelativeSlot(9, flatbuffers.number_types.UOffsetTFlags.py_type(certificateChain), 0)
def RequestExchangeKeysAddCertificateChain(builder, certificateChain):
    """This method is deprecated. Please switch to AddCertificateChain."""
    return AddCertificateChain(builder, certificateChain)
def StartCertificateChainVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def RequestExchangeKeysStartCertificateChainVector(builder, numElems):
    """This method is deprecated. Please switch to Start."""
    return StartCertificateChainVector(builder, numElems)
def End(builder): return builder.EndObject()
def RequestExchangeKeysEnd(builder):
    """This method is deprecated. Please switch to End."""
    return End(builder)