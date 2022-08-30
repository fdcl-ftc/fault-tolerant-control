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
