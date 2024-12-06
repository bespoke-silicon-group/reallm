import abc


################################################################################
#
class Pattern( abc.ABC ):

    def __call__( self, expr ):
        return self.match(expr)

    @abc.abstractmethod
    def match( self, expr ):
        pass


################################################################################
#
class Is( Pattern ):

    def __init__( self, expr_cls, **kwargs ):
        self.expr_cls = expr_cls
        self.kwargs = kwargs

    def match( self, expr ):

        ### Check the expr class matches ###
        if not isinstance(expr, self.expr_cls):
            return False

        ### Go through any additional constraints ###
        for arg,func in self.kwargs.items():

            ### input argument checking
            if arg in expr.get_inputs():
                src_expr = expr.network.lookup_src(getattr(expr,arg))
                if not func(src_expr):
                    return False

            ### attribute argumnet checking
            elif arg in expr.attrs:
                if not func(getattr(expr, arg)):
                    return False

            ### currently don't support output matching
            elif arg in expr.get_outputs():
                raise RuntimeError("{arg} is an output of expr {expr} and cannot be pattern matched.")

            ### unknown arg
            else:
                raise RuntimeError(f"{arg} is not an argument of expr {expr}")

        ### Match found
        return True


################################################################################
#
class Any( Pattern ):

    def match( self, expr ):
        return True


################################################################################
#
class Not( Pattern ):

    def __init__( self, pattern ):
        self.pattern = pattern
    
    def match( self, expr ):
        return not self.pattern.match(expr)


################################################################################
#
class Or( Pattern ):

    def __init__( self, *patterns ):
        self.patterns = patterns
    
    def match( self, expr ):
        for p in self.patterns:
            if p.match(expr):
                return True
        return False

