package com.thoughtworks.compute

import com.thoughtworks.feature.Factory.inject
import com.thoughtworks.feature.{Factory, ImplicitApply}
import com.thoughtworks.compute.NDimensionalAffineTransform.MatrixData

import scala.language.higherKinds

/**
  * @author 杨博 (Yang Bo)
  */
trait Expressions {
  type Category >: this.type <: Expressions

  protected trait TermApi { this: Term =>
    type TermIn[C <: Category] <: C#Term

    type ThisTerm = TermIn[Expressions.this.type]

  }

  type Term <: TermApi

}

object Expressions {

  trait Anonymous extends Any

  object Anonymous {

    implicit def implicitValue[A, Constructor, ImplicitApplied](
        implicit factory: Factory.Aux[(A with Anonymous), Constructor],
        implicitApply: ImplicitApply.Aux[Constructor, ImplicitApplied],
        asImplicitValue: ImplicitApplied <:< (A with Anonymous)
    ): Implicitly[A] = {
      asImplicitValue(implicitApply(factory.newInstance))
    }

  }
  type Implicitly[A] = A with Anonymous

  /**
    * @author 杨博 (Yang Bo)
    */
  trait Values extends Expressions {
    type Category >: this.type <: Values

    protected trait ValueTermApi extends TermApi { this: ValueTerm =>
      type TermIn[C <: Category] <: C#ValueTerm
    }

    /** @template */
    type ValueTerm <: (Term with Any) with ValueTermApi

    protected trait ValueSingletonApi {

      type JvmValue
      type ThisTerm <: ValueTerm

      def literal(value: JvmValue): ThisTerm

      def parameter(id: Any): ThisTerm

    }

    type ValueSingleton <: ValueSingletonApi
  }

  // TODO: Boolean types

  /**
    * @author 杨博 (Yang Bo)
    */
  trait Floats extends Values {
    type Category >: this.type <: Floats

    protected trait FloatTermApi extends ValueTermApi { this: FloatTerm =>
      type TermIn[C <: Category] = C#FloatTerm
      def +(rightHandSide: FloatTerm): FloatTerm
      def -(rightHandSide: FloatTerm): FloatTerm
      def *(rightHandSide: FloatTerm): FloatTerm
      def /(rightHandSide: FloatTerm): FloatTerm
      def %(rightHandSide: FloatTerm): FloatTerm
      def unary_- : FloatTerm
      def unary_+ : FloatTerm
    }

    type FloatTerm <: (ValueTerm with Any) with FloatTermApi

    protected trait FloatSingletonApi extends ValueSingletonApi {
      type JvmValue = Float
      type ThisTerm = FloatTerm
    }

    type FloatSingleton <: (ValueSingleton with Any) with FloatSingletonApi

    @inject
    val float: Implicitly[FloatSingleton]

  }

  /**
    * @author 杨博 (Yang Bo)
    */
  trait Arrays extends Values {
    type Category >: this.type <: Arrays

    protected trait ValueTermApi extends super.ValueTermApi { thisValue: ValueTerm =>

      // TODO: Remove this method
      def fill: ArrayTerm {
        type Element = thisValue.ThisTerm
      }
    }

    override type ValueTerm <: (Term with Any) with ValueTermApi

    protected trait ArrayTermApi extends TermApi { thisArray: ArrayTerm =>
      type TermIn[C <: Category] = C#ArrayTerm {
        type Element = thisArray.Element#TermIn[C]
      }

      type Element <: ValueTerm

      def extract: Element

      def transform(matrix: MatrixData): ThisTerm
    }

    type ArrayTerm <: (Term with Any) with ArrayTermApi

    @inject
    val array: Implicitly[ArrayCompanion]

    protected trait ArrayCompanionApi {

      def parameter[Element0 <: ValueTerm](id: Any, padding: Element0, shape: Array[Int]): ArrayTerm {
        type Element = Element0
      }

    }

    type ArrayCompanion <: ArrayCompanionApi

  }

  /**
    * @author 杨博 (Yang Bo)
    */
  trait FloatArrays extends Floats with Arrays {
    type Category >: this.type <: Floats with Arrays
  }

}
