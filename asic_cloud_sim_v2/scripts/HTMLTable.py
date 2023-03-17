# Highlighting tricks from: https://css-tricks.com/simple-css-row-column-highlighting

HTML_HEADER = '''\
<html>
    <head>
'''

HTML_STYLE_HEADER = '''\
        <style>
            table {
                border: 1px solid black;
                border-collapse: collapse;
                overflow: hidden;
            }

            th {
                color: black;
                background-color: #E5E4E2;
            }

            td {
                table-layout: fixed;
                width: 60px
            }

            td, th {
                text-align: right;
                position: relative;
                font-weight: normal;
                white-space: nowrap;
                padding: 0px 5px 0px 5px;
            }

            tr:hover {
                background-color: lightblue;
            }

            td:hover::after,
            th:hover::after {
                content: "";
                position: absolute;
                background-color: lightblue;
                left: 0;
                top: -5000px;
                height: 10000px;
                width: 100%;
                z-index: -1;
            }
'''

HTML_STYLE_PARETO = '''
            td:nth-child(3n+1) {
                border-right: 3px solid black;
            }
'''

HTML_STYLE_FOOTER = '''
        </style>
    </head>
    <body>
'''

HTML_TABLE_TITLE = '''\
'''

HTML_BODY = '''
'''

HTML_FOOTER = '''
    </body>
</html>
'''

def df2html(df, filename="dataframe", style="Regular", title=HTML_TABLE_TITLE, body=HTML_BODY):
  with open(filename+".html", 'w') as html:
    html.write(HTML_HEADER)
    html.write(HTML_STYLE_HEADER)
    if style == "Pareto":
      html.write(HTML_STYLE_PARETO)
    html.write(HTML_STYLE_FOOTER)
    html.write(title)
    html.write(df.to_html())
    html.write(body)
    html.write(HTML_FOOTER)
