def parse_query(query):
    """
    Parses a multiline SQL query into a one-line string runnable in MS SQL Server.
    Removes --comments.

            Parameters:
                    query (multiline string): A multiline query to parse.

            Returns:
                    parsed_query (str): A one-line query, directly runnable.
    """
    parsedLines = []
    for line in query.splitlines():
        line = line.split('--', 1)[0].strip() # remove comments and whitespaces
        if line: parsedLines.append(line) # include only if non-empty
    return ' '.join(parsedLines)