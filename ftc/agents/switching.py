class Switching():
    def __init__(self, flag=None, criterion=None):
        self.criterion = criterion
        self.flag = flag  # non-initialised flag if `None`

    @property
    def flag(self):
        return self.__flag

    @flag.setter
    def flag(self, flag):
        if self.criterion == None:
            pass
        elif not self.criterion(flag):
            raise ValueError("Invalid flag")
        self.__flag = flag

if __name__ == "__main__":
    criterion = lambda flag: flag >= 0
    switching = Switching(flag=0, criterion=criterion)
    # switching.flag = -1
    # switching.flag = 1
    # switching.flag = 2
    if switching.flag == 1:
        x = 1
    elif switching.flag == 2:
        x = 2
    print(switching.flag)
    print(x)
