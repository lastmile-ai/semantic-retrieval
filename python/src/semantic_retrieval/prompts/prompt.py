class IPrompt:
    async def to_string(self):
        pass


class Prompt(IPrompt):
    def __init__(self, prompt):
        self.prompt = prompt

    async def to_string(self):
        return self.prompt
