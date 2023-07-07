from ftc.registration import register

register(
    id="PPC-DSC",
    entry_point="ftc.controllers.PPC-DSC.ppcdsc:DSCController",
)
register(
    id="LQR",
    entry_point="ftc.controllers.LQR.lqr:LQRController",
)
register(
    id="BLF",
    entry_point="ftc.controllers.BLF.BLF_g:BLFController",
)
register(
    id="BLF-LC62",
    entry_point="ftc.controllers.BLF.BLF_g_LC62:BLFController",
)
register(
    id="Adaptive",
    entry_point="ftc.controllers.Adaptive.adaptive:Adaptive",
)
register(
    id="NDI",
    entry_point="ftc.controllers.NDI.ndi:NDIController",
)
register(
    id="INDI",
    entry_point="ftc.controllers.INDI.indi:INDIController",
)
register(
    id="Flat",
    entry_point="ftc.controllers.Flat.flat:FlatController",
)
