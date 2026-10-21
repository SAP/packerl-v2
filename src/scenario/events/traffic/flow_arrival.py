from src.scenario.events import Event


class FlowArrival(Event):
    """
    Event that represents a traffic flow arrival. This includes the time of the arrival,
    the source and destination nodes, the amount of data to be transmitted, the data rate of the transmission,
    and whether the flow is TCP or UDP.
    """

    def __init__(self, t, src, dst, amount, datarate, is_tcp):
        super().__init__(t)
        self.src: int = src
        self.dst: int = dst
        self.amount: int = amount  # in bytes
        self.datarate: int = datarate  # in bps
        self.is_tcp: bool = is_tcp

    def __str__(self):
        """
        Return a string representation of the event.
        """
        kind_str = "TCP" if self.is_tcp else f"UDP @ {self.datarate}bps"
        return f"TrafficFlowArrival(t={self.t}, ({self.src}, {self.dst}), amount={self.amount}, {kind_str})"

    def apply(self, scenario):
        pass
