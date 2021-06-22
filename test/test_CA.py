from ftc.agents.CA import CA
from ftc.models.multicopter import Multicopter


if __name__ == "__main__":
    def scope(multicopter, ca):
        print(multicopter.mixer.B)
        print(ca.B)

    multicopter = Multicopter()
    ca = CA(multicopter.mixer.B)
    #
    scope(multicopter, ca)
    fault_index = 1
    value = ca.get(fault_index)
    print(value)
    #
    scope(multicopter, ca)

