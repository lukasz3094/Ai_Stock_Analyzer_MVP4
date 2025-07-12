class DataBranchPipeline:
    def load_raw_data(self):
        raise NotImplementedError
    
    def transform(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def run(self):
        self.load_raw_data()
        self.transform()
        self.save()
