import queue


class PeekablePriorityQueue(queue.PriorityQueue):
    def peek(self):
        try:
            with self.mutex:
                return self.queue[0]
        except IndexError:
            raise queue.Empty
