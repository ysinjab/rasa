from tensorflow.python.keras.engine.data_adapter import DataHandler


class CustomDataHandler(DataHandler):
    """Handles iterating over epoch-level `tf.data.Iterator` objects."""

    def enumerate_epochs(self):
        """Yields `(epoch, tf.data.Iterator)`."""
        # TODO
        #  we don't need this anymore once
        #  https://github.com/tensorflow/tensorflow/pull/45338
        #  is merged and released
        with self._truncate_execution_to_epoch():
            data_iterator = iter(self._dataset)
            for epoch in range(self._initial_epoch, self._epochs):
                if self._insufficient_data:  # Set by `catch_stop_iteration`.
                    break
                if self._adapter.should_recreate_iterator():
                    data_iterator = iter(self._dataset)
                    # update number of steps for epoch as we might have an increasing
                    # batch size
                    self._inferred_steps = self._infer_steps(None, self._dataset)
                yield epoch, data_iterator
                self._adapter.on_epoch_end()
